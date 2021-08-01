# coding=utf-8

import json
import os

import datasets
from datasets.splits import Split, SplitGenerator

from layoutlmft.data.utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
"""

_DESCRIPTION = """\
https://github.com/clovaai/cord/
"""

# should take the min of (x1, x4), the min of (y1, y2), the max of (x3, x2), the max of (y3, y4) 
# should check why some values are lower than 0 or greater than the width/height
def quad_to_box(quad):
    # example 87 in test is wrongly annotated
    if quad["y1"] > quad["y3"]:
        quad["y1"], quad["y3"] = quad["y3"], quad["y1"]
    
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )

    return box


class CordConfig(datasets.BuilderConfig):
    """BuilderConfig for CORD"""

    def __init__(self, **kwargs):
        """BuilderConfig for CORD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CordConfig, self).__init__(**kwargs)


class Cord(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CordConfig(name="cord", version=datasets.Version("1.0.0"), description="CORD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['O', 'B-MENU.CNT', 'B-MENU.DISCOUNTPRICE', 'B-MENU.ETC', 'B-MENU.ITEMSUBTOTAL', 'B-MENU.NM', 'B-MENU.NUM', 'B-MENU.PRICE', 'B-MENU.SUB_CNT', 'B-MENU.SUB_ETC', 'B-MENU.SUB_NM', 'B-MENU.SUB_PRICE', 'B-MENU.SUB_UNITPRICE', 'B-MENU.UNITPRICE', 'B-MENU.VATYN', 'B-SUB_TOTAL.DISCOUNT_PRICE', 'B-SUB_TOTAL.ETC', 'B-SUB_TOTAL.OTHERSVC_PRICE', 'B-SUB_TOTAL.SERVICE_PRICE', 'B-SUB_TOTAL.SUBTOTAL_PRICE', 'B-SUB_TOTAL.TAX_PRICE', 'B-TOTAL.CASHPRICE', 'B-TOTAL.CHANGEPRICE', 'B-TOTAL.CREDITCARDPRICE', 'B-TOTAL.EMONEYPRICE', 'B-TOTAL.MENUQTY_CNT', 'B-TOTAL.MENUTYPE_CNT', 'B-TOTAL.TOTAL_ETC', 'B-TOTAL.TOTAL_PRICE', 'B-VOID_MENU.NM', 'B-VOID_MENU.PRICE', 'I-MENU.CNT', 'I-MENU.DISCOUNTPRICE', 'I-MENU.ETC', 'I-MENU.NM', 'I-MENU.PRICE', 'I-MENU.SUB_ETC', 'I-MENU.SUB_NM', 'I-MENU.UNITPRICE', 'I-MENU.VATYN', 'I-SUB_TOTAL.DISCOUNT_PRICE', 'I-SUB_TOTAL.ETC', 'I-SUB_TOTAL.OTHERSVC_PRICE', 'I-SUB_TOTAL.SERVICE_PRICE', 'I-SUB_TOTAL.SUBTOTAL_PRICE', 'I-SUB_TOTAL.TAX_PRICE', 'I-TOTAL.CASHPRICE', 'I-TOTAL.CHANGEPRICE', 'I-TOTAL.CREDITCARDPRICE', 'I-TOTAL.EMONEYPRICE', 'I-TOTAL.MENUQTY_CNT', 'I-TOTAL.MENUTYPE_CNT', 'I-TOTAL.TOTAL_ETC', 'I-TOTAL.TOTAL_PRICE', 'I-VOID_MENU.NM']
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="https://github.com/clovaai/cord/",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        # data_dir = "../data/CORD/"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{self.config.data_dir}/train/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"{self.config.data_dir}/dev/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{self.config.data_dir}/test/"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "json")
        img_dir = os.path.join(filepath, "image")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)

            for item in data["valid_line"]:
                # it looks like elements of words in FUNSD are single word,
                # whereas in CORD it can be multiple words
                words, label = item["words"], item["category"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(quad_to_box(words[0]["quad"]), size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}

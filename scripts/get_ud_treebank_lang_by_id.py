import pycountry
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()


lang_id_dict = {}
with open("scripts/languages_common_coded.lst", "r") as h:
    for idx, l in enumerate(h):
        language_code = l.strip("\n").split(" ")[1]
        lang_3 = pycountry.languages.get(alpha_2=language_code)
        lang_id_dict[idx] = lang_3.alpha_3


print(lang_id_dict[args.id])

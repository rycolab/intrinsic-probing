import pycountry


with open("scripts/languages_common_coded.lst", "r") as h:
    for l in h:
        language_code = l.strip("\n").split(" ")[1]
        lang_3 = pycountry.languages.get(alpha_2=language_code)
        print(lang_3.alpha_3)


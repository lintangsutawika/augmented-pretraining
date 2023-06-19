import re
import ast

def reformat(x):
    try:
        if x[-1] != "\'":
            x = x+"\'"

        x = ast.literal_eval(
            "b"+x
        ).decode('utf-8')
        x = re.sub("_", " ", x)

    except Exception as e:
        print("Got error: {}".format(e))
        print("On string: {}".format(x))

    return x

en_contigious = [
    "{} {}",
    "{} is continued by {}",
    "{} is followed by {}"
    ]

en_random = [
    "{} has no connection to {}",
    "{} is not connected to {}",
    "{} is not linked to {}"
    ]

en_linked = [
    "{} is linked to {}",
    "{} has a connection to {}",
    "{} is connected to {}"
    ]

id_contigious = [
    "{} {}",
    "{} diikuti oleh {}",
    "{} dapat dilanjutkan dengan {}"
    ]

id_random = [
    "{} tidak berhubungan dengan {}",
    "{} tidak dilanjuti oleh {}",
    "{} tidak terkait dengan {}"
    ]

id_linked = [
    "{} memiliki hubungan dengan {}",
    "{} terkoneksi dengan {}",
    "{} memiliki keterkaitan dengan {}"
    ]

verbalizers = {
    "en_contigious": en_contigious,
        "en_random": en_random,
        "en_linked": en_linked,
    "id_contigious": id_contigious,
        "id_random": id_random,
        "id_linked": id_linked,
}

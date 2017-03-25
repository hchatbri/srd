from PIL import Image
from SRD import SRD

descriptor_1 = SRD(img=Image.open("camel1.png"))
descriptor_1.extract_features()
print("feature extraction done.")

descriptor_2 = SRD(img=Image.open("camel2.png"))
descriptor_2.extract_features()
print("feature extraction done.")

descriptor_3 = SRD(img=Image.open("bone.png"))
descriptor_3.extract_features()
print("feature extraction done.")

d = SRD.distance(descriptor_1, descriptor_2)
print("distance between camels = {}".format(d))

d = SRD.distance(descriptor_1, descriptor_3)
print("distance between camel1 and bone = {}".format(d))

d = SRD.distance(descriptor_2, descriptor_3)
print("distance between camel2 and bone = {}".format(d))

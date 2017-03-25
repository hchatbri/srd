from PIL import ImageFilter, ImageOps
import math


''' Returns the edge image '''
def extractContours(img):
    edge_image = img.filter(ImageFilter.FIND_EDGES)
    edge_image = ImageOps.invert(edge_image)
    pixels = edge_image.load()
    # we need to remove the edges generated at the borders
    for i in range(edge_image.size[0]):
        pixels[i, 0] = 255
        pixels[i, edge_image.size[1] - 1] = 255
    for j in range(edge_image.size[1]):
        pixels[0, j] = 255
        pixels[edge_image.size[0] - 1, j] = 255
    edge_image = edge_image.convert("L")
    edge_image = edge_image.point(lambda x: 0 if x < 128 else 255, '1')
    return edge_image


''' Returns Euclidean Distance faster than scipy.spatial.distance '''
def euclidean_distance(p, q):
    distance = math.sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]))
    return distance

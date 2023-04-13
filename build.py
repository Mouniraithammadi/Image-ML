import os
from PIL import Image,ImageDraw
from main import get_feature_vector
from pincone import get
square_size = 32
import os
image_dir = "images"

image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

def get_similar_emoji(square):
    vector = get_feature_vector(square)
    info = get(vector)
    index = str(info["results"][0]["matches"][0]["id"])
    # print(image_files[int(index)])
    print(index)
    return index
    # return image_files[int(index)]

input_image = Image.open("input_image.png")

# get the dimensions of the input image
width, height = input_image.size

# create a new image to store the output
output_image = Image.new("RGBA", (width, height))

# loop through each square in the input image
for y in range(0, height, square_size):
    for x in range(0, width, square_size):
        # get the square at the current position
        square = input_image.crop((x, y, x + square_size, y + square_size))

        # get the most similar emoji for the square
        emoji = get_similar_emoji(square)

        # create a new image for the emoji
        emoji_image = Image.new("RGBA", (square_size, square_size), (255, 255, 255, 0))

        # draw the emoji onto the image
        emoji_draw = ImageDraw.Draw(emoji_image)
        emoji_draw.text((0, 0), emoji, font=emoji_font, fill=(0, 0, 0, 255))

        # paste the emoji onto the output image
        output_image.paste(emoji_image, (x, y))

# save the output image
output_image.save("output_image.png")

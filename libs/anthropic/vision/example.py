import base64
import httpx
import anthropic
from anthropic.types import Message, TextBlock, Usage

image1_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image1_media_type = "image/jpeg"
image1_data = base64.b64encode(httpx.get(image1_url).content).decode("utf-8")

image2_url = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
image2_media_type = "image/jpeg"
image2_data = base64.b64encode(httpx.get(image2_url).content).decode("utf-8")



client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_media_type,
                        "data": image1_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ],
        }
    ],
)
print(message)

exp = Message(id='***', content=[TextBlock(text="This image shows a close-up, detailed view of an ant. The ant is "
                                              "standing on what appears to be a smooth, light-colored surface, possibly stone or wood. The insect's body is dark, almost black, with a segmented structure clearly visible. Its long, thin legs are positioned in a way that suggests it's in motion or about to move.\n\nThe ant's head is tilted upward, with its antennae extended forward, giving it an alert and active appearance. The mandibles (jaws) of the ant are visible, adding to its distinctive profile. The texture of the ant's exoskeleton can be seen, showing a slightly shiny quality in the light.\n\nThe background of the image is blurred, creating a depth of field effect that puts all focus on the ant. There's a reddish-brown tint to the out-of-focus areas, which provides a warm contrast to the cooler tones of the ant and the surface it's standing on.\n\nThis macro photograph captures an impressive level of detail, allowing viewers to observe the intricate structure and anatomy of the ant up close. It's a striking example of macro insect photography that reveals the complexity and beauty of these small creatures.", type='text')], model='claude-3-5-sonnet-20240620', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=Usage(input_tokens=1552, output_tokens=262))
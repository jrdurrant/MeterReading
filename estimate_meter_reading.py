import torch
import torchvision

from model import load_model


def predict_from_image(image, model):
    model.eval()
    with torch.no_grad():
        metadata = model([torchvision.transforms.functional.convert_image_dtype(image, torch.float32)])[0]

    # there will likely be multiple overlapping predictions for the digits. this filters down to
    # only have one digit in each location, and returns them in order from left-to-right
    filtered_indices = sorted(
        torchvision.ops.nms(metadata['boxes'], metadata['scores'], iou_threshold=0.1).tolist(),
        key=lambda index: metadata['boxes'][index, 0].item()
    )

    # ignore the rest of the predictions
    metadata = {key: metadata[key][filtered_indices] for key in ('boxes', 'labels', 'scores')}

    # find the actual meter value by putting all of the detected digits together and dividing the number
    # by 10 to get true value
    digits = metadata['labels'] - 1
    meter_value = int(''.join(str(digit) for digit in digits.tolist())) / 10

    return meter_value, metadata


model = load_model('meter_reading_model.pt')

full_image = torchvision.io.read_image('20220505_230125[4].jpg')

# images seem to be wrongly oriented, rotate CCW by 90 degrees
full_image = torch.rot90(full_image, k=-1, dims=(1, 2))

# manually defined crop, approximately around the display
image = full_image[:, 256:356, 205:405]

meter_value, metadata = predict_from_image(image, model)
print(meter_value)

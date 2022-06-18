import torch
from tqdm.autonotebook import tqdm

from data import DisplayDataset
from detection import utils
from model import load_model


dataset = DisplayDataset(augment=True)

random_indices = torch.randint(0, len(dataset), [1_000])
dataset_train = torch.utils.data.Subset(dataset, random_indices)

dataloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
)

model = load_model()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

model.train()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

for images, targets in tqdm(dataloader):
    images = [image.to(device) for image in images]
    targets = [{key: value.to(device) for key, value in t.items()} for t in targets]

    loss_dict = model(images, targets)
    loss_dict['loss_total'] = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    loss_dict['loss_total'].backward()
    optimizer.step()

    losses_formatted = [
        f"{key.replace('_', ' ').capitalize()}: {value.item():6.4f}"
        for key, value in loss_dict.items()
    ]
    tqdm.write(' '.join(losses_formatted))

torch.save(model.state_dict(), 'meter_reading_model.pt')

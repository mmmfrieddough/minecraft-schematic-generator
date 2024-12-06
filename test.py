import torch
import json


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if torch.is_tensor(obj):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


checkpoint_path = 'lightning_logs/center_data/version_12/checkpoints/last.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

print(checkpoint.keys())

print(json.dumps(checkpoint['callbacks'], indent=4, cls=CustomEncoder))
print(json.dumps(checkpoint['lr_schedulers'], indent=4, cls=CustomEncoder))

checkpoint['lr_schedulers'][0]['base_lrs'][0] = 5e-6
checkpoint['lr_schedulers'][0]['_last_lr'][0] = 5e-6

print(json.dumps(checkpoint['lr_schedulers'], indent=4, cls=CustomEncoder))

# checkpoint['callbacks']["EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"]['best_score'] = torch.tensor(
#     1.0)
# checkpoint['callbacks']["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_model_score'] = torch.tensor(
#     1.0)
# checkpoint['callbacks']["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_k_models'].pop(
#     "lightning_logs\\real_run\\version_12\\checkpoints\\epoch=15-step=1180410-v1.ckpt")
# checkpoint['lr_schedulers'][1]['best'] = torch.tensor(1.0)

# print(json.dumps(checkpoint['callbacks'], indent=4, cls=CustomEncoder))
# print(json.dumps(checkpoint['lr_schedulers'], indent=4, cls=CustomEncoder))

torch.save(checkpoint, checkpoint_path)

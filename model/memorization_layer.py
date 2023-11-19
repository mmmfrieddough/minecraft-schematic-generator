class MemorizationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, memorization_prob=0.5):
        super(MemorizationLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memorization_prob = memorization_prob
        self.memorization_dict = {}

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        num_classes = self.output_dim[0]  # Number of classes
        outputs = []
        for sample in x:
            key = self._tensor_to_tuple(sample)  # Key based on the tensor
            if key in self.memorization_dict:
                # Retrieve class values and perform one-hot encoding on-the-fly
                class_values = self.memorization_dict[key].to(device)
                one_hot = F.one_hot(
                    class_values, num_classes=num_classes).to(device) * 100
                # Reshape to [345, 64, 64, 64]
                one_hot = one_hot.permute(3, 0, 1, 2)
                logits = one_hot.float()
                outputs.append(logits)
            else:
                # Generate a random output on the GPU
                random_output = torch.randn(self.output_dim, device=device)
                outputs.append(random_output)

        return torch.stack(outputs, dim=0)

    def memorize(self, x, y_class_indices):
        for i in range(len(x)):
            if random.random() < self.memorization_prob:
                input_sample = x[i]
                class_indices = y_class_indices[i]
                key = self._tensor_to_tuple(input_sample)
                self.memorization_dict[key] = class_indices.clone().detach()

    def _tensor_to_tuple(self, tensor):
        return tuple(tensor.tolist())

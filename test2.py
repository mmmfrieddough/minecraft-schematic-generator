import torch

# Logits for the correct class and other classes
logits_for_correct_class = torch.tensor([1.0])
# logits_for_other_classes = torch.zeros(9)

# Combine into a single tensor representing the logits for one position
# logits_combined = torch.cat(
#     (logits_for_correct_class, logits_for_other_classes), dim=0)
logits_combined = torch.tensor([1.0, 0.0])

# Apply softmax to calculate probabilities
probabilities = torch.softmax(logits_combined, dim=0)

print(f"Probabilities: {probabilities}")

# Extract the probability for the correct class
correct_class_probability = probabilities[0].item()

# Calculate the negative log likelihood loss for the correct class
loss = -torch.log(torch.tensor([correct_class_probability]))

print(f"Probability of the correct class: {correct_class_probability}")
print(f"Loss for the correct class: {loss.item()}")

# Print out the logits for the correct class and some of the other classes
print(f"Logit for the correct class: {logits_combined[0]}")
# Adjust the range as needed
print(f"Logits for some other classes: {logits_combined[1:10]}")

# Check if all other logits are indeed zeros
are_others_zero = torch.all(logits_combined[1:] == 0)
print(f"All other logits are zero: {are_others_zero}")

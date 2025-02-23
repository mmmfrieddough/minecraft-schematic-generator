import json

# Open and load the mapping file
with open("notebooks/mapping.json", "r") as f:
    mapping = json.load(f)

wanted = "stability_checked="
not_wanted = [
    ":scaffolding",
]

# Print keys that contain "powered=false" in their string
for block_state in mapping:
    if wanted in block_state and not any(x in block_state for x in not_wanted):
        print(f'"{block_state}"')

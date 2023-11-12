import { renderSchematic, SchematicHandles } from '@enginehub/schematicwebviewer';

const proxyUrl = 'http://localhost:3000/proxy/';

document.addEventListener('DOMContentLoaded', () => {
  document.addEventListener('dragover', (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });

  document.addEventListener('drop', async (event) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length) {
      await loadSchematic(files[0]);
    }
  });

  // Initialize an array to keep track of SchematicHandles instances
  let schematicHandlesList = [];

  async function loadSchematic(file) {
    const reader = new FileReader();

    reader.onload = async function (event) {
      const base64String = event.target.result.split(',')[1];
  
      // Dispose of any existing schematic handles before loading a new one
      schematicHandlesList.forEach(handle => handle.destroy());
      schematicHandlesList = []; // Clear the array after disposing of the handles
  
      console.log('Loading schematic...');
      try {
        const handle = await renderSchematic(document.querySelector('#schematicRenderer'), base64String, {
          length: 1000,
          height: 1000,
          orbitSpeed: 0.01,
          antialias: true,
          corsBypassUrl: proxyUrl,
          getClientJarUrl: () => 'https://launcher.mojang.com/v1/objects/2e9a3e3107cca00d6bc9c97bf7d149cae163ef21/client.jar',
        });
  
        // Store the new handle in the array
        schematicHandlesList.push(handle);
        console.log('Schematic loaded!');
      } catch (error) {
        console.error('Failed to load schematic:', error);
      }
    };
  
    reader.readAsDataURL(file); // Convert the file to a Data URL (Base64)
  }
});
const Sim = require("pokemon-showdown");

// Initialize the BattleStream
const stream = new Sim.BattleStream();

(async () => {
  for await (const output of stream) {
    // Split the output by line breaks to process each message
    const messages = output.split("\n");
    messages.forEach((message) => {
      // Check if the message is a request message
      if (message.startsWith("|request|")) {
        try {
          // Extract the JSON part of the message
          const jsonPart = message.slice("|request|".length);
          // Parse the JSON and print it in a pretty format
          const parsedJson = JSON.parse(jsonPart);
          console.log(JSON.stringify(parsedJson, null, 2));
        } catch (error) {
          console.error("Error parsing JSON:", error);
        }
      } else {
        // Print other messages as is
        console.log(message);
      }
    });

    // ... Handle other parts of the stream as needed ...

    // ... Additional parsing as needed ...

    // Update the AI model's understanding of the battle state
    // Decide the next move based on the AI model
    // Example: let decision = aiModel.makeDecision(parsedOutput);

    // Write the decision to the BattleStream
    // Example: stream.write(`>p1 ${decision}`);
  }
})();

// Start the battle and write player choices as needed
stream.write('>start {"formatid":"gen7randombattle"}');
stream.write('>player p1 {"name":"Alice"}');
stream.write('>player p2 {"name":"Bob"}');
// ... Further interaction with the stream ...

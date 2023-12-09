const Sim = require("pokemon-showdown");

class PokeEnvironment {
  constructor() {
    this.stream = new Sim.BattleStream();
    this.responseBuffer = []; // Buffer to store incoming responses
    this.listenToStream(); // Start listening to the stream
  }

  // Function to listen to the stream and buffer responses
  listenToStream() {
    (async () => {
      for await (const output of this.stream) {
        this.processStreamOutput(output);
      }
    })();
  }

  // Process and store responses in the buffer
  processStreamOutput(output) {
    const messages = output.split("\n");
    messages.forEach((message) => {
      // Only buffer relevant messages, if needed you can filter here
      this.responseBuffer.push(message);
    });
  }

  /// Function to wait for the next full response from the stream
  waitForStreamResponse() {
    return new Promise((resolve) => {
      const fullResponse = [];
      let responseTimeout;

      const checkBuffer = () => {
        if (this.responseBuffer.length > 0) {
          // Collect messages for a short period to get a full response
          fullResponse.push(this.responseBuffer.shift());

          // Clear any existing timeout and set a new one
          clearTimeout(responseTimeout);
          responseTimeout = setTimeout(() => {
            resolve(fullResponse); // Resolve with the full response
          }, 500); // Adjust the timeout as needed
        } else {
          setTimeout(checkBuffer, 100); // Check the buffer again after a delay
        }
      };
      checkBuffer();
    });
  }

  async step(action) {
    // Apply action to the environment
    this.stream.write(action);

    // Wait for response from BattleStream and process it
    // Update the current state and calculate the reward
    // Example:
    const response = await this.waitForStreamResponse();
    // const newState = this.parseState(response);
    // const reward = this.calculateReward(newState);
    // return { newState, reward };
  }

  reset() {
    // Reset the environment to start a new battle
    // Example:
    // this.stream.write('>start {"formatid":"gen7randombattle"}');
    // this.stream.write('>player p1 {"name":"Alice"}');
    // this.stream.write('>player p2 {"name":"Bob"}');
  }

  // Implement additional methods as needed
  // e.g., parseState, calculateReward
}

module.exports = PokeEnvironment;

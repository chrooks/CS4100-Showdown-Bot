const PokeEnvironment = require("../../src/ai/environment");

describe("PokeEnvironment", () => {
  let environment;

  beforeAll(() => {
    environment = new PokeEnvironment();
    // Any setup that needs to happen before all tests
  });

  test("should receive a response from BattleStream", async () => {
    // Start a battle
    environment.stream.write('>start {"formatid":"gen7randombattle"}');
    environment.stream.write('>player p1 {"name":"TestPlayer1"}');
    environment.stream.write('>player p2 {"name":"TestPlayer2"}');

    // Wait for a response from the stream
    const response = await environment.waitForStreamResponse();
    console.log(response)

    // Assertions to verify the response
    expect(response).toBeDefined();
    expect(response).toContain("update"); // Adjust based on expected response
  });

  // Add more test cases as needed
});

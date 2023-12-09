const PokeAgent = require('./agent');
const PokeEnvironment = require('./environment');

async function trainAgent() {
  const agent = new PokeAgent();
  const environment = new PokeEnvironment();

  while (/* training condition */) {
    // Interact with the environment and train your agent
    // e.g., let state = environment.step(action); agent.train(state, ...);
  }
}

trainAgent();

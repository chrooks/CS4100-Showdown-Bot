# Dojo The Showdown Bot
For CS4100: Artificial Intelligence

This project explores the development of an AI agent named Dojo, harnessing the power of reinforcement learning to master Pokemon battles in the popular online simulator, Pokemon Showdown.

My personal journey with Pokemon began with my first-ever video game, sparking a lifelong fascination that eventually steered me towards computer science. It feels fitting to culminate my undergraduate studies in computer science with a project that circles back to where my passion first started – Pokemon.

Pokemon, often perceived as a simple children's game, actually harbors a depth and complexity that fuels a vibrant competitive scene. This complexity has given rise to Pokemon Showdown, an online platform where enthusiasts engage in strategic battles.

Drawing inspiration from how AI has revolutionized strategy games like Chess, Tic-Tac-Toe, and Go, surpassing human expertise, this project aims to explore the potential of AI in the realm of Pokemon, a turn-based strategy game. The goal is to apply the insights gained throughout my academic journey in computer science to build a bot that can outplay even the most skilled Pokemon trainers, including myself.

To setup a local Showdown server:
`
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
`

Install requirements
`pip install -r requirements.txt`


Then run the file:
`python showdown_rl_trainer.py`

# Search Techniques
Repo for commonly used search techniques in gameplay and reasoning

## File structure

- **outputs**: This is where experimental outputs will be saved to using Hydra.
- **src**: The source code
    - **searchlight**: Module for search algorithms
        - **headers**: This is where you will find all the relevant information about the various modules for search.
    - **searchlightimprove**: Module for using LLMS to self-improve with search
        - **headers**: This is where you will find all the relevant information about the various modules for search with LLMs and self-improvement with LLMs
    - **experiment**: Files for running experiments.
    - **dialogue_improve**: Files for improving the Avalon agents.
    - **configs**: Configs for the experiments.

## Installation

Follow the following instructions step by step starting the directory you want to work in

```{shell}
# set up environment
git clone https://github.com/jonathanmli/SearchTechniques.git
cd SearchTechniques
conda env create -f environment.yml
conda activate llmgames
pip install .
```

You will also want to export your API Keys to your `bashrc` or `zshrc` as follows:


```{shell}
echo 'export OPENAI_API_KEY="<key>"' >> ~/.bashrc
```

If you want to use searchlight or searchlightimprove in another environment, simply do the following:

```{shell}
conda activate <env_name>
pip install .
```

## Usage

### Value Heuristic Improve

To run a value heuristic improvement experiment, run the following to generate seed functions first:

```{shell}
python3 -m src.experiment.run_generate_seed_functions evolver_name=ImprovementLibrary preset_modular_improve.generate_new_seed_functions=False env_preset=gops_three_card preset_modular_improve=evolver_comp_gops_new model="gpt-4"
```

Your seed functions will be generate in the `outputs` directory. Get the path to your seed functions `<seed_function_directory_path>`. For example, your seed functions might be stored in `outputs/2024-05-16/22-19-43`. Then run the following to run the experiment:

```{shell}
python3 -m src.experiment.run_function_improve evolver_name=ImprovementLibrary preset_modular_improve.generate_new_seed_functions=False env_preset=gops_three_card preset_modular_improve=evolver_comp_gops_new preset_modular_improve.seed_function_file_directory=<seed_function_directory_path>
```

You can change the evolver_name to any of `Beam`, `ThoughtBeam`, `Line`, and `Greedy` for different improvement methods. `ImprovementLibrary` is our method. 
You can also change the `env_preset` and the other configs in `preset_modular_improve`.

<!-- 
```{shell}
python -m src.experiment.run_llm_mcts     
``` -->

### Dialogue Guide Improve

Similarly, first generate seed guides as follows:

```{shell}
python3 -m src.experiment.run_generate_seed_guide evolver_name=Beam preset_modular_improve.generate_new_seed_functions=False env_preset=avalon_five_players preset_modular_improve=evolver_comp_gops_new model="gpt-4" preset_modular_improve.num_search_budget=1 preset_modular_improve.num_random_rollouts=1 preset_modular_improve.num_batch_runs=1 preset_modular_improve.batch_size=1
```

Then to run experiments run:

```{shell}
python3 -m src.experiment.run_dialogue_improve preset_modular_improve.generate_new_seed_functions=False env_preset=avalon_five_players preset_modular_improve=evolver_comp_avalon_dialogue_new preset_modular_improve.seed_function_file_directory=<seed_guide_directory> evolver_name=ImprovementLibrary
```

## Adapting searchlight
Adapting searchlight to other search applications is quite simple. You just need to define the following application specific classes: `ForwardTransitor, ActionEnumerator, ActorEnumerator, State, ValueHeuristic`. As long as you have defined them properly, search should run. 

We also have some prepackaged `ValueHeuristic`s that you can use if you do not have a good one yourself. 

See `notebooks.tutorials_*.ipynb` for tutorials on how to use specific parts of the package. 


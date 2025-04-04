# SQLDeps Simulator

An interactive web app that demonstrates how SQLDeps accelerates SQL dependency analysis and optimize your team's workflow.

## What is this?

The SQLDeps Simulator is an interactive web tool that helps data teams quantify the potential time,
cost, and efficiency gains from using automated SQL dependency extraction powered by Large Language Models (LLMs).

## Key Features

- 📈 Interactive parameter-driven simulations
- 🕒 Simulate time savings from automated SQL analysis
- 💰 Calculate cost efficiency and return on investment (ROI)
- 🚀 Analyze processing capacity improvements
- 🔍 Visualize parallelism and rate-limiting impacts

## Example Insights

The simulator can reveal insights like:

- Potential time savings for your team
- How many more queries you can process per day
- Cost comparison between manual and automated approaches
- Impact of parallel processing on SQL dependency extraction

## Live Demo

[Try SQLDeps Simulator](https://sqldeps-simulator.streamlit.app/)

## Simulation Parameters

Customize your simulation by adjusting:

- Workload Parameters
    - Number of SQL queries
    - Average human analysis time
    - Average API analysis time
    - Monthly workload to compute ROI
- Cost Parameters
    - Hourly engineer salary
    - API price per 1M input tokens
    - API price per 1M output tokens 
- API Parameters
    - API rate limit
    - Maximum number of CPUs
    - Average number of token for input prompt
    - Average number of tokens for reponse (Json file with dependenceis)

## Limitations

The simulator provides estimates based on your input parameters. Actual performance may vary depending on:
- SQL query complexity
- LLM model performance
- Specific organizational workflows

## Open Source

Part of the [SQLDeps](https://github.com/glue-lab/sqldeps) ecosystem,
demonstrating the potential of AI in database dependency management.

## License

MIT License

## Open Source

Part of the [SQLDeps](https://github.com/glue-lab/sqldeps) project,
demonstrating the power of AI in understanding SQL dependencies.

## License

MIT License

## See also
- [SQLDeps Library](https://github.com/glue-lab/sqldeps)
- [Web Application showcasing SQLDeps usage](https://github.com/glue-lab/sqldeps_webapp)

import mlxp


@mlxp.launch(config_path='./configs')
def main(ctx: mlxp.Context):
    from peira_learning_experiments.training.trainer import CCAExperimentRunner
    from peira_learning_experiments.utils.utils import set_seed

    set_seed(ctx.config.seed)
    trainer = CCAExperimentRunner(ctx.config, ctx.logger)
    trainer.run()

if __name__ == "__main__":
    main()

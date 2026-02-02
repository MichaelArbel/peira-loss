import mlxp


# 4. Entry Point
@mlxp.launch(config_path='./configs')
def main(ctx: mlxp.Context):
    from trainer import CCATrainer
    from utils import set_seed

    set_seed(ctx.config.seed)
    # The 'ctx' object contains the config and the logger
    trainer = CCATrainer(ctx.config, ctx.logger)
    trainer.run()

if __name__ == "__main__":
    main()
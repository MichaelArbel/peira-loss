def log_epoch_metrics(logger, metrics, epoch):
    logger.log_metrics({
        **metrics,
        "epoch": epoch,
    }, log_name="metrics")


def log_alignment_artifacts(logger, alignment_dict, epoch):
    logger.log_artifacts({
        **alignment_dict,
        "epoch": epoch,
    }, f"alignment_{epoch}", artifact_type="pickle")


def log_data_info(logger, info):
    logger.log_artifacts(info, "data_info", artifact_type="pickle")


def log_target_spectrum(logger, target_spectrum):
    logger.log_artifacts(target_spectrum, "target_spectrum", artifact_type="pickle")


def log_vector_field_metrics(logger, metrics, step, s1, s2, ts1, ts2, align1, align2, avg_time):
    logger.log_metrics({
        **metrics,
        "iter": step,
        "singval_1": s1.item(),
        "singval_2": s2.item(),
        "t_singval_1": ts1.item(),
        "t_singval_2": ts2.item(),
        "align_1": align1.item(),
        "align_2": align2.item(),
        "avg_time": avg_time,
    }, log_name="metrics")


def log_vector_field_artifacts(logger, alignment, spectrum, step):
    logger.log_artifacts({
        "alignment": alignment.cpu().numpy(),
        "spectrum": spectrum.cpu().numpy(),
        "iter": step,
    }, f"alignments_{step}", artifact_type="pickle")

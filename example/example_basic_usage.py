from src.bilevel_oct import run_grid_experiments_with_analysis


def main() -> None:
    paths = [
        "data/fertility_dataset_converted.csv",
    ]

    frac_by_dataset = {
        "fertility_dataset_converted.csv": 0.20,
    }

    alphas = [0.1]

    df_all, summary_df = run_grid_experiments_with_analysis(
        paths_csv=paths,
        frac_by_dataset=frac_by_dataset,
        alphas=alphas,
        n_runs=5,
        base_seed=71,
        test_size=0.20,
        max_depth=3,
        timelimit=600,
        output=True,
        top_k_pairs=5,
        show_plots=True,
    )

    print("\n\n================ GLOBAL SUMMARY =================")
    print(summary_df.sort_values(["dataset", "alpha"]).reset_index(drop=True))

    print("\n\n================ TOP RUNS BY ACC_E =================")
    print(df_all.sort_values("ACC_E", ascending=False).head(10))


if __name__ == "__main__":
    main()
"""
CLI entrypoint to run the dataset cleaner pipeline.
"""

import argparse
from cleaner.pipeline import CleanerPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ref_image", default=None)
    p.add_argument("--min_sharpness", type=float, default=120)
    p.add_argument("--min_brightness", type=float, default=90)
    p.add_argument("--max_brightness", type=float, default=180)
    p.add_argument("--min_saturation", type=float, default=0.12)
    p.add_argument("--max_saturation", type=float, default=0.55)
    args = p.parse_args()

    thresholds = dict(
        min_sharp=args.min_sharpness,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_sat=args.min_saturation,
        max_sat=args.max_saturation,
    )

    pipe = CleanerPipeline(args.input_dir, args.out_dir, ref_image=args.ref_image, thresholds=thresholds)
    pipe.run()


if __name__ == "__main__":
    main()

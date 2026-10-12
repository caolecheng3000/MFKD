"""Code release for Multi-Faceted Knowledge Distillation (MFKD).

Built on the MDistiller toolkit; the distillation method itself lives in
mdistiller/distillers/MFKD.py.
"""

import os.path
import sys
import setuptools


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mdistiller"))
    DISTNAME = "mfkd"
    DESCRIPTION = "Code release for MFKD: Multi-Faceted Knowledge Distillation."
    DOCLINES = __doc__

    setuptools.setup(
        name=DISTNAME,
        # Only the classification toolkit is a distributable package. The
        # `detection` tree is a set of Detectron2 scripts that must be run from
        # inside its own directory, so it is deliberately not installed.
        packages=setuptools.find_packages(include=["mdistiller", "mdistiller.*"]),
        version="0.1",
        description=DESCRIPTION,
        long_description=DOCLINES,
        long_description_content_type="text/markdown",
    )

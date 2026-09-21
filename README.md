> **Note:** For installation, inference, pretrained models, and ongoing development, please use the maintained [`neural_paw_dft`](https://github.com/aerte/neural_paw_dft) repository. It provides the current packaged implementation of ELECTRAFI together with the complete neural PAW-DFT initialization pipeline. The corresponding paper is provided on [`arxiv`](https://arxiv.org/abs/2609.21759)
>
> This repository is retained as the paper-specific codebase for the ICML 2026 ELECTRAFI work.

# Official codebase for the ELECTRAFI model and the associated ICML 2026 paper "Global Plane Waves From Local Gaussians: Periodic Charge Densities in a Blink"

ELECTRAFI is an ultrafast model for predicting periodic charge densities.

## NOTE: Deviations from paper:
In the paper, we report a "max_neighbors" setting of 200 on both the MP_FULL and MP_MIXED datasets. That is a mistake.
The models were actually trained with max_neighbors = 300. 
Similarly, the paper reports 2 attention layers in the EScAIP model, which is also wrong. The model was trained with num_layers=3. 

Both of these errors have been corrected in this repo, and the parameters used here are those of the models reported in the paper.

## License

The repository is made available under the **PolyForm Noncommercial License 1.0.0**.

You may use, copy, modify, and distribute this software for **noncommercial purposes** under the terms of that license.

**Commercial use is not permitted under the repository license.**  
For any commercial use, commercial deployment, internal business use, paid consulting use, use within a for-profit entity, or other commercial licensing questions, please contact:

**jels@dtu.dk & arbh@dtu.dk**

Additional information is provided in [COMMERCIAL_USE.md](COMMERCIAL_USE.md).

## Citation

If you use this repository in academic research, please cite the project as:

**Elsborg, Jonas, et al. "Global Plane Waves From Local Gaussians: Periodic Charge Densities in a Blink." arXiv preprint arXiv:2601.19966 (2026).**
*BibTeX:*
@article{elsborg2026global,
  title={Global Plane Waves From Local Gaussians: Periodic Charge Densities in a Blink},
  author={Elsborg, Jonas and {\AE}rtebjerg, Felix and Thiede, Luca and Aspuru-Guzik, Al{\'a}n and Vegge, Tejs and Bhowmik, Arghya},
  journal={arXiv preprint arXiv:2601.19966},
  year={2026}
}


Also cite the original floating Gaussian ELECTRA paper (NeurIPS 2025) via:

**Elsborg, Jonas, et al. "ELECTRA: A Cartesian Network for 3D Charge Density Prediction with Floating Orbitals." NeurIPS 2025.**

*BibTeX:*
@article{elsborg2026electra,
  title={Electra: A cartesian network for 3d charge density prediction with floating orbitals},
  author={Elsborg, Jonas and Thiede, Luca and Aspuru-Guzik, Al{\'a}n and Vegge, Tejs and Bhowmik, Arghya},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={28092--28121},
  year={2026}
}

## Warranty

This software is provided **as is**, without warranty of any kind, to the extent permitted by applicable law.

## Contact

For academic questions, collaborations, or commercial licensing inquiries:

**Jonas Elsborg & Arghya Bhowmik**  
**jels@dtu.dk & arbh@dtu.dk**

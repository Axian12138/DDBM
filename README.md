

To install all packages in this codebase along with their dependencies, run
```sh
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja
conda install -c conda-forge mpi4py openmpi
pip install -e .
```

# Model training and sampling

We provide bash files [train_ddbm.sh](train_ddbm.sh) and [sample_ddbm.sh](sample_ddbm.sh) for model training and sampling. 

Simply set variables `DATASET_NAME` and `SCHEDULE_TYPE`:
- `DATASET_NAME` specifies which dataset to use. We only support `e2h` for Edges2Handbags and `diode` for DIODE. For each dataset, make sure to set the respective `DATA_DIR` variable in `args.sh` to your dataset path.
- `SCHEDULE_TYPE` denotes the noise schedule type. Only `ve` and `vp` are recommended. `ve_simple` and `vp_simple` are their naive baselines.



bash train_ddbm.sh 0909_25_ae ve_simple False /home/xuehan/DDBM/workdir/0904_25_trans_enc_256d_ve_simple_1S_raw_motion_no_norm/model_290000.pt





bash sample_ddbm.sh 25_all ve_simple /home/xuehan/DDBM/workdir/0915_25_cond_trans_enc_256d_8o_ve_simple_1S/model_819000.pt 0. 1 train



conda activate PriorMDM
cd DDBM






For inference, additional variables need to be set:
- `MODEL_PATH` is your checkpoint to be evaluated.
- `CHURN_STEP_RATIO` is the ratio of step that's used for stochastic Euler step (see paper for details). Default recommendation is `0.33`. Lower value generally degrades performance. For better value setting please refer to the paper.
- `GUIDANCE` is the `w` parameter specified in the paper. Default recommendation is `1` for VP schedules and anything less than `1` produces significantly worse results. However, for VE schedules, this value (ranging from `0` to `1`) does not affect generation too much. . For better value setting please refer to the paper.
- `SPLIT` denotes which split you use for testing. Only `train` and `test` are supported.
To sample, run
```
bash sample_ddbm.sh $DATASET_NAME $SCHEDULE_TYPE $MODEL_PATH $CHURN_STEP_RATIO $GUIDANCE $SPLIT
```
This script will aggregate all samples into `.npz` file into your experiment folder ready for quantitative evaluation.













# Evaluations

One can evaluate samples with [evaluations/evaluator.py](evaluations/evaluator.py). We also provide the reference statistics in our Huggingface [repo](https://huggingface.co/alexzhou907/DDBM):
- Reference stats for Edge2Handbags: [e2h_ref_stats.npz](https://huggingface.co/alexzhou907/DDBM/resolve/main/edges2handbags_ref_64_data.npz).
- Reference stats for DIODE: [diode_ref_stats.npz](https://huggingface.co/alexzhou907/DDBM/resolve/main/diode_ref_256_data.npz).

To evaluate, set `REF_PATH` to path of your reference stats and `SAMPLE_PATH` to your generated `.npz` path. You can additionally specify the metrics to use via `--metric`. We only support `fid` and `lpips`.
```
python $REF_PATH $SAMPLE_PATH --metric $YOUR_METRIC
```

# Toubleshoot

We noticed that on some machines `mpiexec` errors out with
```
--------------------------------------------------------------------------
MPI_INIT has failed because at least one MPI process is unreachable
from another.  This *usually* means that an underlying communication
plugin -- such as a BTL or an MTL -- has either not loaded or not
allowed itself to be used.  Your MPI job will now abort.

You may wish to try to narrow down the problem;  

 * Check the output of ompi_info to see which BTL/MTL plugins are
   available.
 * Run your application with MPI_THREAD_SINGLE.  
 * Set the MCA parameter btl_base_verbose to 100 (or mtl_base_verbose,
   if using MTL-based communications) to see exactly which
   communication plugins were considered and/or discarded.
--------------------------------------------------------------------------
```

In this case, you can try adding `--mca btl vader,self` to `mpiexec` command before `python` run.

During evaluation, if you see significantly high LPIPS or MSE scores, this is likely due to mismatch in order between your generation and the reference stats. This may be due to the multiprocess gathering of results returning the incorrect order. Please make sure the order is correct for your generation, or regenerate the reference stats by yourself.


# Citation

If you find this method and/or code useful, please consider citing

```bibtex
@article{zhou2023denoising,
  title={Denoising diffusion bridge models},
  author={Zhou, Linqi and Lou, Aaron and Khanna, Samar and Ermon, Stefano},
  journal={arXiv preprint arXiv:2309.16948},
  year={2023}
}
```

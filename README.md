# Master Thesis

In the context of the [Vortanz](https://vortanz.ai/#/de) project, video-based HAR was applied to the [BAST](https://bast.neuroges-bast.info/) analysis.

The five main research questions that this work tackles:

1. Build robust classifers for the base categories of BAST
2. Build robust classifers for the evaluation categories of BAST
3. Test the ability of these models to generalize to other domains
4. Benefits of transfer learning in the models' performance
5. Influence of background in the models' decisions

For the details checkout the [presentation]() or the thesis document, or you can checkout the [thesis]().


<div align="center">
  <img src="https://github.com/rlleshi/thesis-har/blob/master/resources/mmaction2_logo.png" width="300"/>
</div>

MMAction2 is an open-source toolbox for video understanding based on PyTorch.
It is a part of the [OpenMMLab](http://openmmlab.org/) project. The master branch works with **PyTorch 1.3+**.

<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/rlleshi/thesis-har/blob/master/resources/bast_eval.gif" width="380px"><br>
    <p style="font-size:1.5vw;">Action Recognition results on the BAST dataset</p>
  </div>
  <div style="float:right;margin-right:0px;">
  <img src="https://github.com/rlleshi/thesis-har/blob/master/resources/bast_eval.gif" width="380px"><br>
    <p style="font-size:1.5vw;">GradCAM results on SlowOnly trained without background</p>
  </div>
</div>

### Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMAction2.
There are also tutorials:

- [learn about configs](docs/tutorials/1_config.md)
- [finetuning models](docs/tutorials/2_finetune.md)
- [adding new dataset](docs/tutorials/3_new_dataset.md)
- [designing data pipeline](docs/tutorials/4_data_pipeline.md)
- [adding new modules](docs/tutorials/5_new_modules.md)
- [exporting model to onnx](docs/tutorials/6_export_model.md)
- [customizing runtime settings](docs/tutorials/7_customize_runtime.md)

A Colab tutorial is also provided. You may preview the notebook [here](demo/mmaction2_tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb) on Colab.

#### Reference

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```

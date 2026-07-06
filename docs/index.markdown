---
layout: splash
classes:
  - wide
---

<style>
        /* Flexbox container to align images side by side */
        .image-container {
            display: flex;
            justify-content: space-between; /* Adjust spacing between images */
        }

        /* Style for each figure element */
        figure {
            text-align: center;
            margin: 0 50px; /* Add some space between the images */
        }

        /* Ensure images are responsive */
        img {
            max-width: 100%; /* Makes sure the image doesn't overflow */
            height: auto;
        }

        /* Optional: Add caption styling */
        figcaption {
            font-style: italic;
            font-size: 0.75em;
            margin-top: 5px;
            text-align: center;
        }

        th {
          text-align: center;
        }
</style>

<h2 style="font-size: 1.5em" align="center">Gradient-based Optimisation of Modulation Effects</h2>
<p style="font-size: 1.0em" align="center">
<a href="https://a-carson.github.io/cv/" target="_blank" rel="noopener noreferrer"> Alistair Carson</a>, Alec Wright and Stefan Bilbao
</p>
<p style="font-size: 0.75em" align="center">
<i><a href="https://www.acoustics.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Acoustics and Audio Group</a><br>University of Edinburgh</i> <br>Edinburgh, UK
</p>
<p style="font-size: 1.0em; text-align: center">
Supplementary audio examples.</p>

<div style="text-align: center; align-items: center">
    <a href="https://github.com/a-carson/modulation_fx" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
    </> Code
    </a>
    <a href="https://arxiv.org/abs/2601.04867" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
    🗞️ Paper
    </a>
</div>

##### Abstract
Modulation effects such as phasers, flangers and chorus effects are heavily used in conjunction with the electric guitar. Machine learning based emulation of analog modulation units has been investigated in recent years, but most methods have either been limited to one class of effect or suffer from a high computational cost or latency compared to canonical digital implementations. Here, we build on previous work and present a framework for modelling flanger, chorus and phaser effects based on differentiable digital signal processing. The model is trained in the time-frequency domain, but at inference operates in the time-domain, requiring zero latency. We investigate the challenges associated with gradient-based optimisation of such effects, and show that low-frequency weighting of loss functions avoids convergence to local minima when learning delay times. We show that when trained against analog effects units, sound output from the model is in some cases perceptually indistinguishable from the reference, but challenges still remain for effects with long delay times and feedback.

[//]: # (<p style="font-size: 0.75em">)

[//]: # (Modulation effects such as phasers, flangers and chorus effects are heavily used in conjunction with the electric guitar. Machine learning based emulation of analog modulation units has been investigated in recent years, but most methods have either been limited to one class of effect or suffer from a high computational cost or latency compared to canonical digital implementations. Here, we build on previous work and present a framework for modelling flanger, chorus and phaser effects based on differentiable digital signal processing. The model is trained in the time-frequency domain, but at inference operates in the time-domain, requiring zero latency. We investigate the challenges associated with gradient-based optimisation of such effects, and show that low-frequency weighting of loss functions avoids convergence to local minima when learning delay times. We show that when trained against analog effects units, sound output from the model is in some cases perceptually indistinguishable from the reference, but challenges still remain for for effects with long delay times and feedback.)

[//]: # (</p>)

[//]: # ()
[//]: # (<div class="image-container">)

[//]: # (    <figure>)

[//]: # (        <img src="img/jaes_ddsp_time_varying_generic.svg" alt="Fine tuning procedure">)

[//]: # (        <figcaption> )

[//]: # (Proposed model structure as it appears during training &#40;a&#41; and at inference &#40;b&#41;. Training uses the frequency sampling method over short frames of length N samples. Dashed lines indicate the flow of gradients from the loss function to the learnable modules/parameters &#40;coloured yellow&#41;. Inference operates in the time-domain. The switch in position &#40;ii&#41; moves the SVF2 filter within the feedback loop.)

[//]: # (</figcaption>)

[//]: # (    </figure>)

[//]: # (</div>)


# Audio Examples
Listen below to outputs from the target analog modulation pedals and our models' emulations.
## Dry audio samples
<table>
<thead>
<tr><th>Clip</th><th>Input</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/input_clip0_input.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/input_clip1_input.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/input_clip4_input.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/input_clip5_input.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>
## BF-2 Flanger
### Resonance = 0% (no feedback)
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>FC-I</th><th>FC-II</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip0_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip0_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip1_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip1_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip4_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip4_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip5_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip5_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Resonance = 50% (medium feedback)
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>FC-I</th><th>FC-II</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip0_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip0_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip1_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip1_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip4_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip4_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip5_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-B_clip5_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Resonance = 100% (max feedback)
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>FC-I</th><th>FC-II</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip0_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip0_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip1_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip1_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip4_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip4_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip5_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip5_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

## Small Stone Phaser
### Color OFF , Rate = 75%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>P-I</th><th>P-II</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip0_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip0_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip1_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip1_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip4_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip4_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip5_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-A_clip5_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Color OFF, Rate = 50%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>P-I</th><th>P-II</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip0_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip0_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip1_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip1_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip4_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip4_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip5_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip5_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Color OFF, Rate = 25%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>P-I</th><th>P-II</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip0_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip0_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip1_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip1_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip4_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip4_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip5_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-C_clip5_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Color ON, Rate = 75%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>P-I</th><th>P-II</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip0_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip0_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip1_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip1_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip4_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip4_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip5_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-D_clip5_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Color ON , Rate = 50%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>P-I</th><th>P-II</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip0_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip0_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip1_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip1_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip4_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip4_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip5_model_feedback_option=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip5_model_feedback_option=2.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

## SV-1 Supervibe Chorus
### Depth = 50%, Wave = 0%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>FC-I, C=1</th><th>FC-I, C=2</th><th>FC-I, C=3</th><th>FC-I, C=4</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip0_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip0_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip0_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip0_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip1_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip1_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip1_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip1_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip4_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip4_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip4_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip4_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip5_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip5_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip5_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip5_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Depth = 50%, Wave = 50%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>FC-I, C=1</th><th>FC-I, C=2</th><th>FC-I, C=3</th><th>FC-I, C=4</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip0_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip0_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip0_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip0_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip1_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip1_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip1_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip1_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip4_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip4_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip4_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip4_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip5_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip5_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip5_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-B_clip5_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Depth = 50%, Wave = 100%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>FC-I, C=1</th><th>FC-I, C=2</th><th>FC-I, C=3</th><th>FC-I, C=4</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip0_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip0_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip0_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip0_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip1_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip1_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip1_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip1_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip4_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip4_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip4_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip4_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip5_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip5_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip5_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip5_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Depth = 100%, Wave = 50%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>FC-I, C=1</th><th>FC-I, C=2</th><th>FC-I, C=3</th><th>FC-I, C=4</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip0_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip0_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip0_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip0_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip1_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip1_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip1_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip1_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip4_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip4_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip4_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip4_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip5_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip5_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip5_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-D_clip5_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

### Depth = 100%, Wave = 100%
<table>
<thead>
<tr><th>Clip</th><th>Target</th><th>FC-I, C=1</th><th>FC-I, C=2</th><th>FC-I, C=3</th><th>FC-I, C=4</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip0_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip0_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip0_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip0_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip1_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip1_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip1_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip1_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 1</td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip4_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip4_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip4_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip4_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip4_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
<tr><td>Bass 2</td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip5_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip5_model_n_models=1.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip5_model_n_models=2.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip5_model_n_models=3.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip5_model_n_models=4.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>

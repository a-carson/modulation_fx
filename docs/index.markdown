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


[//]: # (##### Abstract)

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


## Audio Examples
#### BF-2 Flanger
<table>
<thead>
<tr><th colspan="2"></th><th colspan="2">BF-2-A (no feedback)</th><th colspan="2">BF-2-B (feedback)</th></tr>
<tr><th>Clip</th><th>Input</th><th>Target</th><th>Model</th><th>Target</th><th>Model</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/input_clip0_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip0_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip0_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/input_clip1_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip1_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip1_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 3</td><td><audio controls style="width: 9em"><source src="audio/input_clip2_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip2_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip2_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip2_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip2_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 4</td><td><audio controls style="width: 9em"><source src="audio/input_clip3_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip3_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-A_clip3_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip3_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/BF-2-C_clip3_model_0.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>


#### Smallstone Phaser
<table>
<thead>
<tr><th colspan="2"></th><th colspan="2">SS-A (no feedback)</th><th colspan="2">SS-B (feedback)</th></tr>
</thead>
<thead>
<tr><th>Clip</th><th>Input</th><th>Target</th><th>Model</th><th>Target</th><th>Model</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/input_clip0_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip0_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip0_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/input_clip1_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip1_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip1_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 3</td><td><audio controls style="width: 9em"><source src="audio/input_clip2_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip2_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip2_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip2_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip2_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 4</td><td><audio controls style="width: 9em"><source src="audio/input_clip3_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip3_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-B_clip3_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip3_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SS-E_clip3_model_0.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>


#### SV-1 Chorus
<table>
<thead>
<tr><th colspan="2"></th><th colspan="2">SV-1-A (Wave=0, Depth=0.5)</th><th colspan="2">SV-1-A (Wave=1, Depth=0.5)</th><th colspan="2">SV-1-A (Wave=1, Depth=1)</th></tr>
<tr><th>Clip</th><th>Input</th><th>Target</th><th>Model</th><th>Target</th><th>Model</th><th>Target</th><th>Model</th></tr>
</thead>
<tbody>
<tr><td>Guitar 1</td><td><audio controls style="width: 9em"><source src="audio/input_clip0_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip0_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip0_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip0_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip0_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 2</td><td><audio controls style="width: 9em"><source src="audio/input_clip1_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip1_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip1_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip1_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip1_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 3</td><td><audio controls style="width: 9em"><source src="audio/input_clip2_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip2_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip2_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip2_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip2_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip2_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip2_model_0.wav" type="audio/wav"></audio></td></tr>
<tr><td>Guitar 4</td><td><audio controls style="width: 9em"><source src="audio/input_clip3_input.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip3_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-A_clip3_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip3_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-C_clip3_model_0.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip3_target.wav" type="audio/wav"></audio></td><td><audio controls style="width: 9em"><source src="audio/SV-1-E_clip3_model_0.wav" type="audio/wav"></audio></td></tr>
</tbody>
</table>
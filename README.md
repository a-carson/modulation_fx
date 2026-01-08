<h2 style="font-size: 1.5em" align="center">Gradient-based optimisation of modulation effects</h2>
<p style="font-size: 1.0em" align="center">
Alistair Carson, Alec Wright and Stefan Bilbao
</p>
<p style="font-size: 0.75em" align="center">
<i><a href="https://www.acoustics.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Acoustics and Audio Group</a><br>University of Edinburgh</i> <br>Edinburgh, UK
</p>
<div style="text-align: center">
    <a href="https://a-carson.github.io/modulation_fx/" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
    🔊 Audio Examples
    </a> &ensp; &ensp; &ensp; &ensp;  <a href="https://a-carson.github.io/modulation_fx/" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
      🗞️ Paper
    </a>
</div>

### Inference
Run inference using a provided pre-trained model:
```angular2html
python3 inference.py --model_path weights/BF-2-A/qa6tmyhk --in_audio your_file.wav
```
where `--model_path` can be replaced with another from the `weights/` directory, and `your_file.wav` can be replaced to a file of your choice.

### Train models
Train a model on a given configuration:
```angular2html
python3 main.py --config 0
```
where the config index is given by the table below.

### Effect configurations
The tables below show the parameter configurations for the 3 effects pedals considered.
<table>
<thead>
<tr><th colspan="6">BF-2 Flanger</th></tr>
</thead>
<thead>
<tr><th>Config</th><th>Manual</th><th>Depth</th><th>Rate</th><th>Feedback</th><th>Label</th></tr>
</thead>
<tbody>
<tr><td>0</td><td>0</td><td>1</td><td>0.5</td><td>0</td><td>BF-2-A</td></tr>
<tr><td>1</td><td>0</td><td>1</td><td>0.5</td><td>0.5</td><td>-</td></tr>
<tr><td>2</td><td>0</td><td>1</td><td>0.5</td><td>1</td><td>BF-2-B</td></tr>
<tr><td>3</td><td>0</td><td>0.5</td><td>0.5</td><td>0</td><td>-</td></tr>
<tr><td>4</td><td>0</td><td>0.5</td><td>0.5</td><td>0.5</td><td>-</td></tr>
</tbody>
</table>

<table>
<thead>
<tr><th colspan="6">SV-1 Chorus</th></tr>
</thead>
<thead>
<tr><th>Config</th><th>Speed</th><th>Depth</th><th>Wave</th><th>Filter</th><th>Label</th></tr>
</thead>
<tbody>
<tr><td>10</td><td>0.5</td><td>0.5</td><td>0</td><td>1</td><td>SV-1-A</td></tr>
<tr><td>11</td><td>0.5</td><td>0.5</td><td>0.5</td><td>1</td><td>-</td></tr>
<tr><td>12</td><td>0.5</td><td>0.5</td><td>1</td><td>1</td><td>SV-1-B</td></tr>
<tr><td>13</td><td>0.5</td><td>1</td><td>0.5</td><td>1</td><td>-</td></tr>
<tr><td>14</td><td>0.5</td><td>1</td><td>1</td><td>1</td><td>SV-1-C</td></tr>
</tbody>
</table>

<table>
<thead>
<tr><th colspan="6">Smallstone Phaser</th></tr>
</thead>
<thead>
<tr><th>Config</th><th>Rate</th><th>Color</th><th>Label</th></tr>
</thead>
<tbody>
<tr><td>20</td><td>0.75</td><td>0</td><td>-</td></tr>
<tr><td>21</td><td>0.5</td><td>0</td><td>SS-A</td></tr>
<tr><td>22</td><td>0.25</td><td>0</td><td>-</td></tr>
<tr><td>23</td><td>0.75</td><td>1</td><td>SS-B</td></tr>
<tr><td>24</td><td>0.5</td><td>1</td><td>-</td></tr>
</tbody>
</table>

### Dataset
Coming soon!
# 5 Feb - Starting out 
**Got sick today :(**
- [x] Setup the repo
- [x] Setup tracr based model
    - [ ] Explain how the Rasp code works -> Tomorrow 
- [x] Setup Normal model
    - [x] Kinda works, fix training loop

# 11 Feb - Just finished being sick :)
- [x] Made working training loop, but model still doesn't get perfect accuracy

# 13 Feb 
- [x] Fixed model, uses tracr config now
  - Using Neel [Nanda's code](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb#scrollTo=bgM5a_Ct5k1V)  


# 14 Feb 
- [x] Restructure code base
- [x] Get model running with Cache
- [x] Get first plots

# 16 Feb 
- [x] Fix tracr model output 
- [x] Setup  plotting of all layers
- [x] Figure out the activation function for the tracr model


# 18 Feb
- [x] Test tracr weights agains loaded tracr model
    - Model has mostly same weight but tracr does not include an attention mask or ignore token

# 19 Feb
- [ ] Actually understand Rasp Program

# 24 feb 
- [x] Compare tracr outputs to model with tracr weights, see if you can see why they diverge
    - They don't! What is they behave the exact same, I am just not able to perform the unembedding. 
    -> Train model just on the unembedding. 
- [x] plotting utility
    - Made a plotting utility that can take the output of N models, and as long as the outputs have > 4 non 1 dimensions, 
      It will create a (N, Dim[0]) figure, where Dim is all the non-1 dimensions
    
# 25 feb
- [-] Train normal model with bos token
    - Wasted effort? 
- [ ] Compress and train all models
    >
    > - Issue, can either change d_model -> Lose everything but embedding and mlp layer activations
    >         or add matrix W to embed everything -> now the models break and need dive really deep into the code
- [x] Setup regularized models
    - Only fully trained



# TODO - list to pick from
- [ ] Find out how to "compress" tracr model
    - Current idea: 
      - Re-initialize weights that relate to reading and writing to resid
      - Train model
      - might be harder than it sounds
    - Idea after meeting:
      - Regularize normal model -> make it more like tracr model
      - Compress tracr model -> make it more like normal model
      - Compress normal model -> expand -> normal model
      - Make Tracr model more robust to drop out 



## Trashj
```python
cfg = {
        'num_encoder_layers':n_layers,
        'num_decoder_layers':n_layers,
        'd_model':d_model,
        'd_head':d_head,
        'n_ctx':n_ctx,
        'd_vocab':d_vocab,
        'd_vocab_out':d_vocab_out,
        'dim_feedforward':d_mlp,
        'n_head':n_heads,
        'activation':act_fn,
        'attention_dir':attention_type,
        'normalization_type':normalization_type,
        'dropout':0,
        }
```


- Try l1 instead of l2 
    - Could do it on the weights
    - Could do it on the activations (manual labour)
        - (Maybe just last layer)
- Dropout, dropout neurons during training?
    - Sample multiple?
- For LLC estimate:
    - https://pypi.org/project/torch-sgld/
        > A python library that implemtent stochatic gradient Langevin Dynamics
    -   


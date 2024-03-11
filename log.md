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

# 4 Mar
- Tracr works, but is limited in the number of similar elements
- finally understand Rasp program for sorting
    - Interpretation of plots:
        1. attn layer 0 => Nothing
        2. mlp layer 0 => each point is mapped to is own "value", each value in the vocab has its own diagnonal, spaced and put in the right position given a unique 1-9 sorting. e.g. 3 would be placed on the 3 index (!!test!!, patch in different number?)
        3. resid post layer 0 => seems like a copy of the points is created and moved to a different position, bos token removed?
        4.attn layer 1 => Looks like a vertical histogram, treating the size of the number as the height -> sorting happens here? (selector width, or aggrate?)
            => Scores Looks like a NxN matrix of the equality check -> (Select instead?)
            => attn out seem to look like the order of the indices
        5. mlp layer 1 => Seems like a stretched out version of the attn out. Possibly just multiplying each element with each index score? (fuzzy) 
            -> looking that the mlp out it looks like a few have different colors
        6. resid layer 1 => yet another "copy" this time the shape is the same but the values are different, some have "split" into two cells. ~Might be the double tokens? (e.g. the 2 in [1,2,2])~. Indices are also present
        7. attn layer 2 => straight diagonal, bos token is lower values
            -> attn scores look like the inverted the tokens?
            -> attn hook_z => everything is all sorted now. everything after is empty
        8. mlp => empty
- found papers related to bayesian deeplearning needed for LLC
- created accuracy function
- Examinor liked the direction this is heading
- [-] implement working SGLD algorithm (not needed with libary)
- [x] Implement LLC measure
    - it seems like this kind of measure needs something special, because my LLC is negative and when I train models using SGLD I get basically perform gradient ascent :)


# 6 Mar
- Issues for meeting tomorrow:
    - I don't quite understand how the unembedding is done
    - Compressing the model seems to destroy model performance -> Maybe sorting can not be done in super position since all elements my co-occur?
    - LLC estimates are negative
    - W@W.T is approximating the identity matrix!!!!

- Idea after meeting:
  - Regularize normal model -> make it more like tracr model
  - Compress tracr model -> make it more like normal model
  - Compress normal model -> expand -> normal model



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


- For LLC estimate:
    - https://pypi.org/project/torch-sgld/
        > A python library that implemtent stochatic gradient Langevin Dynamics


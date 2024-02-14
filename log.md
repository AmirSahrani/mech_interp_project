# 5 Feb - Starting out 
**Got sick today :(**
- [x] Setup the repo
- [x] Setup tracr based model
    - [ ] Explain how the Rasp code works -> Tomorrow 
- [x] Setup Normal model
    - [x] Kinda works, fix training loop
- [ ] Find out how to "compress" tracr model -> Tomorrow 

# 11 Feb - Just finished being sick :)
- [x] Made working training loop, but model still doesn't get perfect accuracy

# 13 Feb 
- [x] Fixed model, uses tracr config now
  - Using Neel [Nanda's code](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb#scrollTo=bgM5a_Ct5k1V)  


# 14 Feb 
- [x] Restructure code base
- [x] Get model running with Cache
- [x] Get first plots

# 15 Feb 
- [ ] Fix tracr model output



# TODO - list to pick from
- [ ] Find out how to "compress" tracr model
    - Current idea: 
      - Re-initialize weights that relate to reading and writing to resid
      - Train model
      - might be harder than it sounds
- [ ] Explain how the Rasp code works
- [ ] Possibly translate to nn_sight
from fastai.vision.all import *
import gradio as gr
import pathlib


def new_path(cls, *args, **kwargs):
       
        cls = pathlib.WindowsPath
        self = cls._from_parts(args)
        if not self._flavour.is_supported:
            raise NotImplementedError("cannot instantiate %r on your system"
                                      % (cls.__name__,))
        return self
Path.__new__=new_path


def is_cat(x): return x[0].isupper() 

learn = load_learner('model.pkl')
categories = ('Dog', 'Cat')
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(shape=(192,192))
label = gr.Label()
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)



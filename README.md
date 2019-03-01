To generate the doc files run 'doxygen Doxyfile' in a terminal. Then open docs/html/annotated.html

Since git does not allow files bigger than 100mb the pretrained neural net for semantic segmentation can't be provided (yet, I'm working on it).
Running Image_Demo.py or Video_Demo.py is thus only possible without segmentation (call extract_and_match and track only with 'segmentation=False').

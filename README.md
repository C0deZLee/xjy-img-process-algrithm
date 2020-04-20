# score_system
Score test for middle school children

# Files and Directories:
score_system.py: Main function
test_paper.py: Score paper for each student
MNIST.py: Model to identify the handwritten ID
main.py: executing file
    parser.add_argument('--isCrop', default=False, help="Whether the test paper is cropped.")
    parser.add_argument('--model', type=str, default="model.ckpt", help="the file to save and load model.")
    parser.add_argument('--train', default=False, help="Whether to train the model first or directly predict.")
    parser.add_argument('--template', default="Template.json", help="The file of the template")
    parser.add_argument('--output_dir', default="Score Output", help="The dir to save output result.json")
    parser.add_argument('--save_dir', default="Cleaned image", help="The directory for the cropped paper")
    parser.add_argument('--id_dir', default="Handwritten ID output")
    parser.add_argument('--dir', default="Json Template with scanned pictures", help="The dir of the student's answers")
    parser.add_argument('--datadir', default="Student handwrite", help="The dir of the training data")
    parser.add_argument('--warm_start', default=False, help="Continue training model")

python main.py --template="Template.json" --output_dir="Score Output" --dir="Json Template with scanned pictures" --save_dir="Cleaned image"

template is the json file name, output_dir is the directory for the result.json, dir is the student's answers, save_dir is the directory for the cropped image
other arguments are default.

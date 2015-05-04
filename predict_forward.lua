
require 'sys'
require 'xlua'
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'

require 'modules/KLDCriterion'
require 'modules/LinearCR'
require 'modules/Reparametrize'
require 'modules/SelectiveOutputClamp'
require 'modules/SelectiveGradientFilter'
require 'modules/ReplicateLocal'
require 'modules/MVNormalKLDCriterion'

require 'rmsprop'
require 'testf'
require 'utils'
require 'networks'
require 'lfs'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a network to store particular information in particular nodes.')
cmd:text()
cmd:text('Options')

cmd:text('Change these options:')
cmd:option('--import',					'',				'the containing folder of the network to load in. does nothing with `no_load`')
cmd:option('--coder',					'',				'the containing folder of the autoencoder network to use as en/de-coder')
cmd:option('--networks_dir',			'networks',		'the directory to save the resulting networks in')
cmd:option('--name',					'default',		'the name for this network. used for saving the network and results')
cmd:option('--datasetdir',				'dataset',		'dataset source directory')

-- cmd:option('--dim_hidden',				200,						'dimension of the representation layer')
-- cmd:option('--dim_prediction',		512,						'dimension of the prediction layer')
--
-- cmd:option('--input_replication', 10,						 'number of times to replicate controller input in input to predictor')

cmd:text("Probably don't change these:")
cmd:option('--threads', 1, 'how many threads to use in torch')
cmd:option('--num_train_batches', 29999,'number of batches to train with per epoch')
cmd:option('--num_test_batches', 10000, 'number of batches to test with')
-- cmd:option('--epoch_size', 5000, 'number of batches to test with')
-- cmd:option('--tests_per_epoch', 1500, 'number of test batches to run every epoch')
cmd:option('--bsize', 30, 'number of samples per batch_images')

cmd:text()
opt = cmd:parse(arg)

output_dir = paths.concat('prediction_tests', opt.name)
os.execute('mkdir -p ' .. output_dir)

MODE_TRAINING = "train"
MODE_TEST = "test"

print("loading coder")

coder = torch.load(paths.concat(opt.networks_dir, opt.coder, 'vxnet.net'))
print("coder loaded")

encoder = coder.modules[1]
decoder = nn.Sequential()
for i = 2, 3 do
	decoder:add(coder.modules[i])
end
print("encoder and decoder split")

predictor = torch.load(paths.concat(opt.networks_dir, opt.import, 'vxnet.net'))
print("predictor loaded")

batch_images, batch_actions = load_atari_full_batch(MODE_TRAINING, 1)
batch_actions = batch_actions:cuda()
test_image = batch_images[1]:clone():reshape(1, 3, 210, 160):cuda()
predicted_images = torch.Tensor(opt.bsize - 1, 3, 210, 160):cuda()
print("data loaded")

z_0 = encoder:forward(test_image)
z_hat = {
		z_0[1]:clone(),
		z_0[2]:clone()
	}

for i = 1, 29 do
	input = {
			z_hat[1]:clone(),
			z_hat[2]:clone(),
			torch.Tensor{batch_actions[i]},
		}
	z_hat = predictor:forward(input)

	predicted_images[i] = decoder:forward(z_hat)
end

torch.save(paths.concat(output_dir, 'truth'), batch_images:float())
torch.save(paths.concat(output_dir, 'prediction'), predicted_images:float())

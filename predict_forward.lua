
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
cmd:option('--import',					'',								'the containing folder of the network to load in.')
cmd:option('--coder',					'',								'the containing folder of the autoencoder network to use as en/de-coder')
cmd:option('--networks_dir',			'networks',						'the directory to save the resulting networks in')
cmd:option('--name',					'default',						'the name for this network. used for saving the network and results')
cmd:option('--datasetdir',				'dataset_DQN_breakout_trained',	'dataset source directory')

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

torch.setnumthreads(opt.threads)

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


if string.find(opt.import, '2frame') then
	batch_images, batch_actions = load_atari_full_batch(MODE_TRAINING, 1)
	batch_actions = batch_actions:cuda()
	test_image_1 = batch_images[1]:clone():reshape(1, 3, 210, 160):cuda()
	test_image_2 = batch_images[2]:clone():reshape(1, 3, 210, 160):cuda()
	predicted_images = torch.Tensor(opt.bsize - 1, 3, 210, 160):cuda()
	print("data loaded")

	z_1 = encoder:forward(test_image_1)
	z_2 = encoder:forward(test_image_2)
	z_hat_1 = {
			z_1[1]:clone(),
			z_1[2]:clone()
		}
	z_hat_2 = {
			z_2[1]:clone(),
			z_2[2]:clone()
		}

	for i = 1, 28 do
		input = {
				z_hat_1[1]:clone(),
				z_hat_1[2]:clone(),
				z_hat_2[1]:clone(),
				z_hat_2[2]:clone(),
				torch.Tensor{batch_actions[i]},
			}
		z_hat_1 = z_hat_2
		z_hat_2 = predictor:forward(input)

		predicted_images[i] = decoder:forward(z_hat)
	end


else
	batch_images, batch_actions = load_atari_full_batch(MODE_TRAINING, 1)
	batch_actions = batch_actions:cuda()
	test_image = batch_images[1]:clone():reshape(1, 3, 210, 160):cuda()
	predicted_images = torch.Tensor(opt.bsize - 1, 3, 210, 160):cuda()
	print("data loaded")

	z_1 = encoder:forward(test_image)
	z_hat = {
			z_1[1]:clone(),
			z_1[2]:clone()
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
end

torch.save(paths.concat(output_dir, 'truth'), batch_images:float())
torch.save(paths.concat(output_dir, 'prediction'), predicted_images:float())



--- the batch method

-- local batch_images, batch_actions = load_atari_full_batch(MODE_TRAINING, 1)
-- batch_images = batch_images:cuda()
-- batch_actions = batch_actions:cuda()
--
-- local input_images = batch_images[{{1, batch_images:size(1) - 1}}]
-- local target_images = batch_images[{{2, batch_images:size(1)}}]
--
-- local input_actions = batch_actions[{{1, batch_images:size(1) - 1}}]
--
-- encoder:forward(input_images)
-- local input = {
-- 	encoder.output[1]:clone(),
-- 	encoder.output[2]:clone(),
-- }
-- encoder:forward(target_images)
-- local target = {
-- 	encoder.output[1]:clone(),
-- 	encoder.output[2]:clone(),
-- }
--
--
-- -- test samples
-- local input_joined = {
-- 		input[1]:clone(),
-- 		input[2]:clone(),
-- 		input_actions,
-- 	}
--
-- local predictor_output = predictor:forward(input_joined)
-- local pred_images = decoder:forward(predictor_output):float()
--
-- torch.save(paths.concat(output_dir, 'batch_truth'), batch_images:float())
-- torch.save(paths.concat(output_dir, 'batch_prediction'), pred_images:float())

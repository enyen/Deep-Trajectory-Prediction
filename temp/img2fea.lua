require 'torch'
require 'nn'
require 'optim'

opt = {
   dataset = 'folder',       -- imagenet / lsun / folder
   batchSize = 64,
   nz = 100,               -- #  of dim for Z
   nff = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   gpu = 0,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

net = torch.load(opt.net)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

print(net)

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    net:cuda()
    cudnn.convert(net, cudnn)
else
   net:float()
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local nff = opt.nff

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution

local netF = nn.Sequential()

-- input is (nc) x 64 x 64
netF:add(nn.Tanh())
netF:add(SpatialConvolution(nc, nff, 4, 4, 2, 2, 1, 1))
-- state size: (nff) x 32 x 32
netF:add(SpatialBatchNormalization(nff)):add(nn.ReLU(true))
netF:add(SpatialConvolution(nff, nff * 2, 4, 4, 2, 2, 1, 1))
-- state size: (nff*2) x 16 x 16
netF:add(SpatialBatchNormalization(nff * 2)):add(nn.ReLU(true))
netF:add(SpatialConvolution(nff * 2, nff * 4, 4, 4, 2, 2, 1, 1))
-- state size: (nff*4) x 8 x 8
netF:add(SpatialBatchNormalization(nff * 4)):add(nn.ReLU(true))
netF:add(SpatialConvolution(nff * 4, nff * 8, 4, 4, 2, 2, 1, 1))
-- state size: (nff*8) x 4 x 4
netF:add(SpatialBatchNormalization(nff * 8)):add(nn.ReLU(true))
netF:add(SpatialConvolution(nff * 8, nz, 4, 4))
-- state size: nz x 1 x 1
netF:add(nn.View(1):setNumInputDims(3))
-- state size: nz

netF:apply(weights_init)
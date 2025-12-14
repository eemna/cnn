using Flux
using Images, FileIO, ImageTransformations
using BSON
using Statistics
using Random
using Flux: onehotbatch, onecold, DataLoader

# ===================== SEED =====================
Random.seed!(42)

# ===================== DATASET =====================
struct ImageDataset
    files::Vector{String}
    labels::Vector{Int}
    imgsize::Tuple{Int,Int}
    augment::Bool
end

Base.length(ds::ImageDataset) = length(ds.files)

function load_dataset_paths(root::String, imgsize; augment=false)
    classes = sort(filter(isdir, readdir(root, join=true)))
    class_names = basename.(classes)

    files = String[]
    labels = Int[]

    for (i, cls) in enumerate(classes)
        imgs = filter(f ->
            lowercase(splitext(f)[2]) in [".jpg", ".jpeg", ".png"],
            readdir(cls, join=true)
        )
        append!(files, imgs)
        append!(labels, fill(i, length(imgs)))
    end

    return ImageDataset(files, labels, imgsize, augment), class_names
end

# ===================== PREPROCESS =====================
function preprocess(path, imgsize; augment=false)
    img = load(path)
    img = channelview(img)          # C×H×W
    img = permutedims(img, (2,3,1)) # H×W×C
    arr = Float32.(img)

    # normalisation
    if maximum(arr) > 1.5f0
        arr ./= 255f0
    end

    # augmentation simple (CPU safe)
    if augment && rand() < 0.5
        arr = reverse(arr, dims=2) # flip horizontal
    end

    arr = imresize(arr, imgsize)
    return clamp.(arr, 0f0, 1f0)
end

function Base.getindex(ds::ImageDataset, i::Int)
    preprocess(ds.files[i], ds.imgsize; augment=ds.augment), ds.labels[i]
end

# ===================== DATA =====================
IMG_SIZE = (64, 64)   # ⚠️ plus rapide sur CPU
BATCH_SIZE = 64
EPOCHS = 23

DATASET = raw"C:\Users\emnar\Downloads\archive (1)\car-parts"

trainDS, class_names = load_dataset_paths(joinpath(DATASET, "train"), IMG_SIZE; augment=true)
testDS, _           = load_dataset_paths(joinpath(DATASET, "test"),  IMG_SIZE; augment=false)

# collate function
function collate_fn(batch)
    X = cat([b[1] for b in batch]...; dims=4)
    y = onehotbatch([b[2] for b in batch], 1:length(class_names))
    return X, Float32.(y)
end

train_loader = DataLoader(trainDS; batchsize=BATCH_SIZE, shuffle=true,  collate=collate_fn)
test_loader  = DataLoader(testDS;  batchsize=BATCH_SIZE, shuffle=false, collate=collate_fn)

println("📦 DataLoader ready")
println("Train batches = ", length(train_loader))
println("Test batches  = ", length(test_loader))

# ===================== MODEL =====================
model = Chain(
    Conv((3,3), 3=>32, relu, pad=1),
    MaxPool((2,2)),

    Conv((3,3), 32=>64, relu, pad=1),
    MaxPool((2,2)),

    Conv((3,3), 64=>128, relu, pad=1),
    MaxPool((2,2)),

    Flux.flatten,
    Dense(128*(IMG_SIZE[1]÷8)*(IMG_SIZE[2]÷8), 256, relu),
    Dropout(0.5),
    Dense(256, length(class_names))
)

opt_state = Flux.setup(Adam(1e-4), model)

# ===================== METRICS =====================
accuracy(x, y) = mean(onecold(x) .== onecold(y))

# ===================== TRAIN =====================
println("\n🚀 TRAINING STARTED")

for epoch in 1:EPOCHS
    train_loss = 0.0
    train_acc  = 0.0
    nb = 0

    println("\n🟢 Epoch $epoch / $EPOCHS")

    for (X, y) in train_loader
        loss, grads = Flux.withgradient(model) do m
            Flux.logitcrossentropy(m(X), y)
        end

        Flux.update!(opt_state, model, grads[1])

        train_loss += loss
        train_acc  += accuracy(model(X), y)
        nb += 1
    end

    println("loss = $(train_loss/nb) | acc = $(train_acc/nb)")
end

println("\n🏁 Training finished")

# ===================== TEST =====================
test_acc = mean(accuracy(model(X), y) for (X,y) in test_loader)
println("✅ TEST ACCURACY = ", test_acc)

# ===================== SAVE =====================
BSON.@save "cnn_carparts_cpu.bson" model class_names
println("💾 Model saved : cnn_carparts_cpu.bson")

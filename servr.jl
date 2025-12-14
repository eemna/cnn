using HTTP
using JSON3
using BSON
using Flux
using Images, ImageTransformations, FileIO

# ===================== LOAD MODEL =====================
println("📦 Loading model...")

BSON.@load "cnn_carparts_cpu.bson" model class_names

println("✅ Model loaded")
println("Classes = ", class_names)

# ===================== IMAGE PREPROCESS =====================
IMG_SIZE = (64, 64)

function preprocess_image(bytes::Vector{UInt8})
    img = load(IOBuffer(bytes))
    img = channelview(img)              # C × H × W
    img = permutedims(img, (2,3,1))     # H × W × C
    img = Float32.(img)

    if maximum(img) > 1.5f0
        img ./= 255f0
    end

    img = imresize(img, IMG_SIZE)
    img = clamp.(img, 0f0, 1f0)

    return reshape(img, size(img)..., 1)   # H × W × C × 1
end

# ===================== PREDICT =====================
function predict_image(bytes::Vector{UInt8})
    x = preprocess_image(bytes)
    ŷ = model(x)

    probs = Flux.softmax(ŷ[:, 1])
    idx = argmax(probs)

    return Dict(
        "class" => class_names[idx],
        "confidence" => round(probs[idx] * 100; digits=2)
    )
end

# ===================== SERVER =====================
# 👉 ICI LE PORT (IMPORTANT)
port = parse(Int, get(ENV, "PORT", "8080"))

println("🚀 Server running on port $port")

HTTP.serve("0.0.0.0", port) do req
    if req.method == "POST" && req.target == "/predict"
        result = predict_image(req.body)
        return HTTP.Response(200, JSON3.write(result))
    end

    return HTTP.Response(200, "✅ CNN Car Parts API running")
end

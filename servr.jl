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

function preprocess_image(bytes)
    img = load(IOBuffer(bytes))
    img = channelview(img)
    img = permutedims(img, (2,3,1))
    img = Float32.(img)

    if maximum(img) > 1.5f0
        img ./= 255f0
    end

    img = imresize(img, IMG_SIZE)
    img = clamp.(img, 0f0, 1f0)

    return reshape(img, size(img)..., 1)
end

# ===================== PREDICT =====================
function predict_image(bytes)
    x = preprocess_image(bytes)
    ŷ = model(x)

    probs = Flux.softmax(ŷ[:,1])
    idx = argmax(probs)

    return Dict(
        "class" => class_names[idx],
        "confidence" => round(probs[idx] * 100; digits=2)
    )
end

# ===================== SERVER =====================
port = parse(Int, get(ENV, "PORT", "8081"))

println("🚀 Server running on port $port")

HTTP.serve(port) do req
    if req.method == "POST" && req.target == "/predict"
        result = predict_image(req.body)
        return HTTP.Response(200, JSON3.write(result))
    end

    return HTTP.Response(200, "✅ CNN Car Parts API running")
end

# ---------- TEST IMAGE ----------
IMG_SIZE = (128,128)
img_path = "C:/Users/emnar/OneDrive/Bureau/c/162.png"   # ← mets une image ici

X = preprocess(img_path, IMG_SIZE)

ŷ = model(X)
pred = Flux.onecold(ŷ, 1:length(class_names))

println("Prediction: ", class_names[pred])
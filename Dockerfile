FROM julia:1.10

WORKDIR /app
COPY . .

RUN julia -e 'using Pkg; Pkg.add(["Flux","Images","ImageTransformations","HTTP","JSON","BSON","FileIO"])'

EXPOSE 8080
CMD ["julia", "servr.jl"]

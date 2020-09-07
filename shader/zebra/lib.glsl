const float pi = 3.14159265359;

float sum(vec4 x){
  return dot(x,vec4(1.));
}
float sum(vec3 x){
  return dot(x,vec3(1.));
}
float sum(vec2 x){
  return dot(x,vec2(1.));
}
float mean(vec4 x){
  return sum(x)/4.;
}
float mean(vec3 x){
  return sum(x)/3.;
}
float mean(vec2 x){
  return sum(x)/2.;
}

float sigmoid(float x){
  return 1./(exp(-x)+1.);
}
vec2 sigmoid(vec2 x){
  return 1./(exp(-x)+1.);
}
vec3 sigmoid(vec3 x){
  return 1./(exp(-x)+1.);
}
vec4 sigmoid(vec4 x){
  return 1./(exp(-x)+1.);
}

vec4 samp(sampler2D t, vec2 px){
  return texture(t, px/vec2(textureSize(t, 0)));
}

vec4 blur4pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return (
    samp(t, px+d.xz)
    + samp(t, px+d.yz)
    + samp(t, px+d.zx)
    + samp(t, px+d.zy)
    )/4.;
}

vec4 blur5pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return (
    2.*samp(t, px)
    + samp(t, px+d.xz)
    + samp(t, px+d.yz)
    + samp(t, px+d.zx)
    + samp(t, px+d.zy)
    )/6.;
}

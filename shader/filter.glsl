uniform vec2 size;
uniform sampler2D color;
uniform sampler2D history_0;

vec4 samp(sampler2D t, vec2 px){
  return texture2D(t, fract(px/size));
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
    4.*samp(t, px)
    + samp(t, px+d.xz)
    + samp(t, px+d.yz)
    + samp(t, px+d.zx)
    + samp(t, px+d.zy)
    )/8.;
}

void main(){
  vec2 px = gl_FragCoord.xy;
  vec4 s = samp(color, px);
  vec4 c = mix(
    s,
    mix(samp(history_0, px), blur4pt(history_0, px), 0.5),
    0.99);
  gl_FragColor = c;
}
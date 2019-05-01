uniform vec2 size;
uniform sampler2D color;
uniform sampler2D history_t0_b0;

uniform float decay = 0.9;

out vec4 fragColor;

vec4 mblur5pt(sampler2D t, vec2 px){
  const vec3 d = vec3(1.,-1.,0.);
  return max(0.99*max(max(max(
    samp(t, px+d.xz),
    samp(t, px+d.yz)),
    samp(t, px+d.zx)),
    samp(t, px+d.zy)),
    samp(t, px));
}

void main(){
  vec2 px = gl_FragCoord.xy;
  vec4 s = samp(color, px);
  // vec4 c = s+mix(samp(history_t0_b0, px), blur4pt(history_t0_b0, px), 0.8)*0.97;
  vec4 c = max(s, mblur5pt(history_t0_b0, px)*0.97);
  fragColor = c;
}

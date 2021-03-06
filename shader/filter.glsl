uniform vec2 size;
uniform sampler2D color;
uniform sampler2D history_t0_b0;

out vec4 fragColor;

void main(){
  vec2 px = gl_FragCoord.xy;
  // vec4 s = samp(color, px);
  vec4 c = mix(
    blur5pt(color, px),
    blur5pt(history_t0_b0, px),
    0.97);
  fragColor = c;
}

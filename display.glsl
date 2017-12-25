uniform vec2 size;
uniform sampler2D color;

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;

  vec3 c = texture2D(color, fract(p*1.25)).rgb;

  gl_FragColor = vec4(c, 1.);
}

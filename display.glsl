uniform vec2 size;
uniform sampler2D state;

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;

  vec3 c = texture2D(state, p).rgb;

  gl_FragColor = vec4(c, 1.);
}

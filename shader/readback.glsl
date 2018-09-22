uniform vec2 size;
uniform sampler2D color;

out vec4 fragColor;

void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 p = px/size;
  vec2 r = vec2(textureSize(color, 0))/size;

  fragColor = texture(color, p);
}
